#lang racket
(define (digitSum s)
  (for/sum ([c (in-string s)] #:when (char-upper-case? c)) (char->integer c)))
