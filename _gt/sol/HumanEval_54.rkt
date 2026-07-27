#lang racket
(define (same_chars s0 s1)
  (equal? (list->set (string->list s0)) (list->set (string->list s1))))
