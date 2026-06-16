#lang racket
(define (string_xor a b)
  (list->string
   (map (lambda (ca cb) (if (char=? ca cb) #\0 #\1))
        (string->list a) (string->list b))))
