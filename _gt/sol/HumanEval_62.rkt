#lang racket
(define (derivative xs)
  (for/list ([c (in-list xs)] [i (in-naturals)] #:when (> i 0)) (* c i)))
