#lang racket
(define (is_palindrome s)
  (string=? s (list->string (reverse (string->list s)))))
